Nuclei Segmentation with U-Net

This is a learning project. I had not worked on image segmentation before and wanted to understand it properly, so I built a small working model instead of only reading about it. The code is kept simple on purpose, and the notes below are the things I actually learned while doing it.

The task

Classification says what is in a picture. Detection draws a box around it. Segmentation labels every single pixel. So the output of this model is not a number — it is a picture the same size as the input, where each pixel says "nucleus" or "background". That output picture is called a mask.

The data

Data Science Bowl 2018 — microscope images of cell nuclei, with a hand-drawn mask for every nucleus.

I picked this dataset because the nuclei are small, there are many per image, and they touch each other. Separating two touching objects turns out to be much harder than finding them, and that is the interesting part of the problem.

The model

A U-Net with a ResNet-34 encoder, via segmentation-models-pytorch.

A U-Net squeezes the image down through layers until it understands what is in it, then expands it back up to full size to say where. The path down and back up is what gives it the name. The important trick is the skip connections: the fine detail from the way down is handed across to the way up — by the time the image has been squeezed small, the model knows what it is looking at but has lost the exact edges.

Scoring

Accuracy is the wrong measure here. If nuclei cover 10% of an image, a model that answers "background" for every pixel is 90% accurate and completely useless. So the model is scored with IoU (the overlap between the predicted mask and the true mask, divided by their combined area) and Dice.

Notes as I go

(filled in as I work through it)

