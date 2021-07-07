from sagemaker_sklearn_extension.decomposition import RobustPCA
from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.feature_extraction.text import MultiColumnTfidfVectorizer
from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Given a list of column names and target column name, Header can return the index
# for given column name
HEADER = Header(
    column_names=['sentiment', 'review_body'], target_column_name='sentiment'
)


def build_feature_transform():
    """ Returns the model definition representing feature processing."""

    # These features can be parsed as natural language.

    text = HEADER.as_feature_indices(['review_body'])

    text_processors = Pipeline(
        steps=[
            (
                'multicolumntfidfvectorizer',
                MultiColumnTfidfVectorizer(
                    max_df=0.99,
                    min_df=0.0021,
                    analyzer='char_wb',
                    max_features=10000
                )
            )
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[('text_processing', text_processors, text)]
    )

    return Pipeline(
        steps=[
            ('column_transformer',
             column_transformer), ('robustpca', RobustPCA(n_components=5)),
            ('robuststandardscaler', RobustStandardScaler())
        ]
    )


def build_label_transform():
    """Returns the model definition representing feature processing."""

    return RobustLabelEncoder(labels=['-1', '0', '1'])
