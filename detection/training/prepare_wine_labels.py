"""Prepare the actual full-label COCO boxes, preserving the single wine-labels class."""
from prepare_text_fields import main, ROOT

if __name__ == '__main__':
    main(source='Wine Labels', output=ROOT/'wine_labels_yolo')
