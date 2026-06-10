@echo off
echo 启动后端 API (端口 8000)...
start "Backend API" cmd /k "conda activate school && python run_api.py"

echo 启动前端开发服务器 (端口 3000)...
start "Frontend Dev" cmd /k "cd web && npm run dev"

echo 服务已启动，请勿关闭这两个窗口。
pause