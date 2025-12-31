from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(X_data):
    var_num = X_data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    var_cat = X_data.select_dtypes(include=["object", "category"]).columns.tolist()
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), var_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), var_cat)
    ])