import cv2
import numpy as np
import torch
import torchvision.transforms as T
#import matplotlib.pyplot as plt

def apply_mask(image, mask):
    masked_image = np.copy(image)
    for c in range(3):
        masked_image[:, :, c] = np.where(mask == 1, image[:, :, c], 0)
    return masked_image

def segment_person(image, model, scale_factor=0.5):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Downsample the image
    small_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    input_image = transform(small_image).unsqueeze(0)
    output = model(input_image)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = (output_predictions == 15)

    # Upsample the mask back to the original size
    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

#def on_key(event, cap):
#    if event.key == 'q':
#        cap.release()
       # plt.close()

def main():
    # Load the DeepLabv3 model
    model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Set the webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get the webcam frame size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load and resize the background image
    background = cv2.imread('images/bg.jpg')
    background = cv2.resize(background, (width, height))

    #plt.ion()
    #fig, ax = plt.subplots()

    # Set the callback for key events
    #fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, cap))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Segment the person from the frame
       # mask = segment_person(frame, model)

        # Apply the mask to the frame
       # masked_frame = apply_mask(frame, mask)

        # Replace the background
       # result = np.where(mask[..., None], masked_frame, background)

        # Show the result
        cv2.imshow("BGRemove2",frame)
       # ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
       # plt.pause(0.01)
       # plt.draw()

       # if not plt.get_fignums():
       #     break

    cap.release()
   # plt.close()

if __name__ == '__main__':
    main()