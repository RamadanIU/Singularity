@echo off
chcp 65001 >nul
cd /d "%~dp0"
python slr_net_trainer_chat.py --corpus corpus_movie_subtitles_and_logic.txt --epochs 6 --save slr_net_trained.npz
pause
