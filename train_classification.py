from ultralytics import YOLO
import torch
import argparse

parser = argparse.ArgumentParser(description="Yolo Training")
parser.add_argument("--src",help="source directory of the datasets",required=True)
parser.add_argument("--epochs",help="number of epochs",default=20,type=int)
parser.add_argument("--imgsz",help="image size",default=640,type=int)
parser.add_argument("--name",help="directory to save the model",default="")
args = parser.parse_args()

if __name__ == "__main__":
    model = YOLO('yolo11n-cls.pt')
    if args.name == "":
        results = model.train(
            data=args.src,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            workers=3,
            batch=32
        )
    else:
        results = model.train(
            data=args.src,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            workers=3,
            batch=32,
            name=args.name
        )