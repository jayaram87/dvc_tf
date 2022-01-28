from setuptools import setup


## edit below variables as per your requirements -
REPO_NAME = "dvc_tf"
AUTHOR_USER_NAME = "jayaram87"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = [
    "dvc",
    "tqdm",
    "tensorflow",
    "joblib"
]


setup(
    name=SRC_REPO,
    version="0.0.3",
    author=AUTHOR_USER_NAME,
    description="A small package for DVC",
    long_description='DVC TF sample',
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="jayaramraja1987@gmail.com",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)