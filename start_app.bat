@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Wissen AF3 Explorer v3.1 启动脚本 - uv 虚拟环境版
set "VENV_DIR=F:\uv\envs\alphafold_tools"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PIP=%VENV_DIR%\Scripts\pip.exe"
set "STREAMLIT=%VENV_DIR%\Scripts\streamlit.exe"

title Wissen AF3 Explorer v3.1 - 自动化自愈启动引擎
echo ===========
echo          Wissen AF3 Explorer - AlphaFold 3 二次分析一站式平台 v3.1
echo ===========
echo.
echo [环境检查] Python: %PYTHON%
echo [环境检查] 正在自动检测并对齐 Python 运行时依赖库...
echo [环境检查] 正在执行: %PIP% install -r requirements.txt
echo -----------------------------------------------------------------------
"%PIP%" install -r requirements.txt
if %errorlevel% neq 0 (
    echo [警告] 依赖库安装遇到问题，继续尝试启动...
)
echo -----------------------------------------------------------------------
echo [环境检查] 依赖库校验对齐完成。
echo.
echo [系统指引] 正在初始化大分子分析看板的前端响应上下文...
echo [系统指引] 正在拉起 Streamlit Web 服务器 (http://localhost:8501)...
echo.
echo -----------------------------------------------------------------------
"%PYTHON%" -m streamlit run main_app.py --server.port 8501
if %errorlevel% neq 0 goto error_handler
goto end

:error_handler
echo.
echo [致命错误] 服务拉起遭遇未预期阻断！
echo   1. 检查虚拟环境: %VENV_DIR%
echo   2. 端口 8501 是否被占用？
echo   3. 手动执行: "%PIP%" install -r requirements.txt
echo -----------------------------------------------------------------------
pause

:end