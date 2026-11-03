@echo off
chcp 65001 >nul
cd /d "%~dp0"
python slr_net_trainer_chat.py --load slr_net_trained.npz
pause
