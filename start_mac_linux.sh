#!/usr/bin/env bash
# Запуск из любой папки: переходим туда, где лежит скрипт
cd "$(dirname "$0")" || exit 1
if [ "$1" = "--instant" ]; then
  # Мгновенный чат с уже обученными весами
  python3 slr_net_trainer_chat.py --load slr_net_trained.npz
else
  # Обучение чат-прототипа на корпусе и запуск чата
  python3 slr_net_trainer_chat.py --corpus corpus_movie_subtitles_and_logic.txt --epochs 6 --save slr_net_trained.npz
fi
