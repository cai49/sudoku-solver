if [ -z "$( ls -A '/path/to/dir' )" ]; then
   python digit_classifier.py
fi

python solver.py --model .\build\classifier.h5 --image ./test_img.jpg
