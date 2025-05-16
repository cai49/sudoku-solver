@echo off

dir /b /s /a ".\build" | findstr .>nul || (
    call python digit_classifier.py
)
call python solver.py --model .\build\classifier.keras --image .\test_images\test_img_flat_ss.jpg