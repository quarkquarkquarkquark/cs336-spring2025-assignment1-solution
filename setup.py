from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

# 1. 定义扩展模块
# Extension 对象描述了一个需要编译的独立模块。
# - 第一个参数 "finder" 是你编译后想用的模块名，即 `import finder`。
# - 第二个参数是一个源文件列表。这里我们只有一个源文件 'finder.pyx'。
ext_modules = [
    Extension(
        "finder",
        ["./cs336_basics/finder.pyx"],
    )
]

# 2. 调用 setup 函数
# `cythonize` 是 Cython 的魔法函数，它会处理 Extension 对象，
# 将 .pyx 文件转换成 .c 文件，然后让 setuptools 继续编译 .c 文件。
setup(
    name="Cython Finder",
    ext_modules=cythonize(ext_modules)
)