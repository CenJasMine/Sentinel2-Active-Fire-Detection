import os
import glob
import pickle
import logging
import numpy as np
from itertools import product

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# —— 日志配置 ——
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 类别名称映射
CLASS_NAMES = {
    0: "亮屋顶", 1: "火", 2: "自然噪声", 3: "过火地", 4: "人工噪声"
}

def load_txt_folder(folder_path):
    """加载文件夹下所有 TXT 文件"""
    cols = (
        ["ID", "category", "lon", "lat", "img_id"] +
        [f"Band{i}" for i in range(1, 9)] +
        ["BandB8A"] +
        [f"Band{i}" for i in [9, 10, 11, 12]] +
        ["solar_zenithN", "solar_azimuthN"] +
        ["NDVIMax","NDVIMin"]+
        ["R1Max","R1Min", "R2Max","R2Min","R3Max","R3Min"]+
        ["B1Max","B1Min","B2Max","B2Min"]+
        ["W1Max","W1Min", "BandSimMax", "BandSimMin"]+  # 新增列
        ["R4Max","R4Min", "R5Max", "R5Min"]+
        ["B3Max","B3Min", "B4Max", "B4Min"]
    )

    logger.info(f"正在加载文件夹: {folder_path}")
    all_files = glob.glob(os.path.join(folder_path, "*.txt"))

    if not all_files:
        logger.error(f"在 {folder_path} 中没有找到任何TXT文件")
        raise FileNotFoundError(f"文件夹 {folder_path} 中没有找到TXT文件")

    dfs = []
    for fp in all_files:
        try:
            df = pd.read_csv(fp, sep=",", header=None, names=cols)
            df = df.dropna(subset=["category"])
            df["category"] = df["category"].astype(int)
            dfs.append(df)
        except Exception as e:
            logger.error(f"加载文件 {fp} 时出错: {str(e)}")
            continue

    if not dfs:
        raise ValueError("没有成功加载任何数据文件")

    data = pd.concat(dfs, ignore_index=True)
    data = data.drop(columns=["ID", "lon", "lat", "img_id"])

    logger.info(f"总共加载了 {len(data)} 行数据，类别分布: {data['category'].value_counts().to_dict()}")
    return data


def train_model(train_folder, val_folder, param_grid=None, use_gpu=False, use_val_for_final_training=True):
    """训练模型并实现惩罚机制，可选择是否将验证集加入最终训练"""
    # 1. 加载数据
    df_train = load_txt_folder(train_folder)
    df_val = load_txt_folder(val_folder)

    # 去掉太阳角度特征
    feature_cols = [c for c in df_train.columns if c not in ["category","solar_zenithN", "solar_azimuthN"]]
    X_train, y_train_orig = df_train[feature_cols], df_train["category"]
    X_val, y_val_orig = df_val[feature_cols], df_val["category"]

    # 2. 标签编码
    le = LabelEncoder()
    le.fit(np.unique(np.concatenate([y_train_orig, y_val_orig])))
    y_train = le.transform(y_train_orig)
    y_val = le.transform(y_val_orig)

    # 记录映射关系
    class_mapping = {original: encoded for original, encoded in zip(le.classes_, range(len(le.classes_)))}
    inverse_mapping = {encoded: original for original, encoded in class_mapping.items()}
    logger.info(f"类别映射: {class_mapping}")
    logger.info(f"训练集类别分布: {np.bincount(y_train)}")
    logger.info(f"验证集类别分布: {np.bincount(y_val)}")

    # 3. 设置惩罚机制 - 重点关注火点类别
    fire_encoded_class = class_mapping.get(1)  # 火点类别
    roof_encoded_class = class_mapping.get(0)  # 亮屋顶类别

    # 创建类别权重
    class_weights = {c: 1.0 for c in class_mapping.values()}
    class_weights[fire_encoded_class] = 2.0  # 火点的权重
    class_weights[roof_encoded_class] = 1.0  # 亮屋顶的权重

    # 生成样本权重（仅用于超参数搜索阶段）
    sample_weights_train = np.array([class_weights[y] for y in y_train])
    logger.info(f"类别权重设置: {class_weights}")

    # 4. 基础参数设置
    base_params = {
        "objective": "multi:softprob",
        "num_class": len(class_mapping),
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "verbosity": 1
    }

    # 5. 超参数搜索或直接训练（仅使用训练集进行验证）
    best_params = None
    best_model = None

    if param_grid:
        logger.info("开始超参数搜索（使用训练集+验证集进行评估）")
        best_score = -1
        best_fire_recall = -1

        if "scale_pos_weight" in param_grid:
            del param_grid["scale_pos_weight"]

        param_combinations = [
            dict(zip(param_grid.keys(), values))
            for values in product(*param_grid.values())
        ]

        logger.info(f"共有 {len(param_combinations)} 种参数组合")

        for i, params in enumerate(param_combinations):
            logger.info(f"训练参数组合 {i+1}/{len(param_combinations)}: {params}")
            try:
                current_params = {**base_params, **params}
                model = xgb.XGBClassifier(**current_params)

                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)

                if fire_encoded_class is not None:
                    fire_recall = recall_score(y_val, y_pred, labels=[fire_encoded_class], average='micro')
                    logger.info(f"参数组合 {i+1} 验证集准确率: {score:.4f}, 火点召回率: {fire_recall:.4f}")

                    cm = confusion_matrix(y_val, y_pred)
                    if roof_encoded_class is not None:
                        fire_as_roof = cm[fire_encoded_class, roof_encoded_class]
                        total_fire = np.sum(cm[fire_encoded_class, :])
                        if total_fire > 0:
                            fire_as_roof_rate = fire_as_roof / total_fire
                            logger.info(f"火点误分类为亮屋顶比例: {fire_as_roof_rate:.4f} ({fire_as_roof}/{total_fire})")

                    weighted_score = 0.3 * score + 0.7 * fire_recall

                    if weighted_score > best_score or (weighted_score == best_score and fire_recall > best_fire_recall):
                        best_score = weighted_score
                        best_fire_recall = fire_recall
                        best_params = params
                        logger.info(f"找到新的最佳模型，加权分数: {weighted_score:.4f}, 火点召回率: {fire_recall:.4f}")
                else:
                    logger.info(f"参数组合 {i+1} 验证集准确率: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_params = params
                        logger.info(f"找到新的最佳模型，准确率: {best_score:.4f}")

            except Exception as e:
                logger.error(f"参数组合 {i+1} 训练失败: {str(e)}")
                continue

        if best_params is None:
            logger.warning("所有参数组合都训练失败，使用默认参数")
            best_params = {
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_estimators": 200
            }

        logger.info(f"超参数搜索完成，最佳参数: {best_params}")
    else:
        best_params = {
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 200
        }
        logger.info("使用默认参数")

    # 6. 使用最佳参数训练最终模型
    final_params = {**base_params, **best_params}
    best_model = xgb.XGBClassifier(**final_params)

    if use_val_for_final_training:
        logger.info("使用训练集+验证集训练最终模型")
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_combined = np.concatenate([y_train, y_val])
        sample_weights_combined = np.array([class_weights[y] for y in y_combined])

        logger.info(f"合并后数据集大小: {len(X_combined)}")
        logger.info(f"合并后类别分布: {np.bincount(y_combined)}")

        best_model.fit(
            X_combined, y_combined,
            sample_weight=sample_weights_combined,
            verbose=True
        )

        logger.info("最终模型在原验证集上的表现:")
        y_pred_final = best_model.predict(X_val)
        acc_final = accuracy_score(y_val, y_pred_final)
        logger.info(f"验证集准确率(非独立验证): {acc_final:.4f}")


        if fire_encoded_class is not None:
            fire_recall_final = recall_score(y_val, y_pred_final, labels=[fire_encoded_class], average='micro')
            logger.info(f"火点召回率: {fire_recall_final:.4f}")

        cm_final = confusion_matrix(y_val, y_pred_final)
        logger.info(f"最终混淆矩阵:\n{cm_final}")

        if fire_encoded_class is not None and roof_encoded_class is not None:
            fire_as_roof_final = cm_final[fire_encoded_class, roof_encoded_class]
            total_fire_final = np.sum(cm_final[fire_encoded_class, :])
            if total_fire_final > 0:
                fire_as_roof_rate_final = fire_as_roof_final / total_fire_final
                logger.info(f"火点误分类为亮屋顶比例: {fire_as_roof_rate_final:.4f} ({fire_as_roof_final}/{total_fire_final})")

    else:
        logger.info("仅使用训练集训练最终模型")
        best_model.fit(
            X_train, y_train,
            sample_weight=sample_weights_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

        y_pred = best_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        logger.info(f"验证集准确率: {acc:.4f}")

        cm = confusion_matrix(y_val, y_pred)
        logger.info(f"混淆矩阵:\n{cm}")

        if fire_encoded_class is not None and roof_encoded_class is not None:
            fire_as_roof = cm[fire_encoded_class, roof_encoded_class]
            total_fire = np.sum(cm[fire_encoded_class, :])
            if total_fire > 0:
                fire_as_roof_rate = fire_as_roof / total_fire
                logger.info(f"火点误分类为亮屋顶比例: {fire_as_roof_rate:.4f} ({fire_as_roof}/{total_fire})")

    best_model.label_encoder_ = le
    best_model.class_mapping_ = class_mapping
    best_model.inverse_mapping_ = inverse_mapping

    return best_model


def save_model(model, path):
    """保存模型到本地"""
    if model is not None:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"模型已保存到 {path}")
    else:
        logger.warning(f"模型为None，无法保存到 {path}")


def predict_with_original_labels(model, X):
    """使用模型预测，并将结果转换回原始类别标签"""
    y_pred = model.predict(X)
    y_pred_orig = model.label_encoder_.inverse_transform(y_pred.astype(int))
    return y_pred_orig


if __name__ == "__main__":
    try:
        param_grid = {
            "learning_rate": [0.2, 0.3, 0.4],
            "max_depth": [4, 5, 6],
            "n_estimators": [400, 500, 600],
            "subsample": [0.7, 0.75, 0.8],
            "colsample_bytree": [0.7, 0.75, 0.8],
            "min_child_weight": [0.1, 0.5, 0.75]
        }

        logger.info("开始训练过程")
        model = train_model(
            train_folder=r"Path to training data folder",
            val_folder=r"Path to validation data folder",
            param_grid=param_grid,
            use_gpu=False,
            use_val_for_final_training=True
        )

        save_model(model, r"F:\CJM\straw_burning\4NewIndex\Model\训练_预测_Model\20260316_1.pkl")
        logger.info("训练完成，已保存最终模型")

    except Exception as e:
        logger.error(f"主程序执行出错: {str(e)}", exc_info=True)
