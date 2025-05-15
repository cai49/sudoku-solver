@echo off

dir /b /s /a ".\build" | findstr .>nul || (
    call python digit_classifier.py
)
call python solver.py --model .\build\classifier.h5 --image .\test_img.jpg