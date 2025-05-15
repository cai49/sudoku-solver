if [ -z "$( ls -A './build' )" ]; then
   python digit_classifier.py
fi

python solver.py --model build/classifier.keras --image ./test_img.jpg
