@echo off
cd /d D:\VLLM\vllm-gfx906
git status
git log --oneline 2>&1 | head -10
