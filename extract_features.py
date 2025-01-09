import numpy as np
from scipy.ndimage import gaussian_filter

def imageProcessing(img, typeOfImage='digit'):
    original = np.array([[float(pixel) for pixel in row] for row in img])
    blur = 0.7 if typeOfImage == 'digit' else 1.0
    smooth = gaussian_filter(original, sigma=blur)
    pixelsCells = 7 if typeOfImage == 'digit' else 10
    imageDimensions = (28, 28) if typeOfImage == 'digit' else (70, 60)
    gradientY = np.pad(np.diff(smooth, axis=0), ((0,1), (0,0)), typeOfImage='edge')
    gradientX = np.pad(np.diff(smooth, axis=1), ((0,0), (0,1)), typeOfImage='edge')
    magnitude = np.sqrt(gradientX**2 + gradientY**2)
    height, width = imageDimensions
    cellHeight = height // pixelsCells
    cellWidth = width // pixelsCells
    features = []

    for i in range(pixelsCells):
        for j in range(pixelsCells):
            xStart = j * cellWidth
            xEnd = min((j + 1) * cellWidth, width)
            yStart = i * cellHeight
            yEnd = min((i + 1) * cellHeight, height)
            chunk = original[yStart:yEnd, xStart:xEnd]
            edgeChunks = magnitude[yStart:yEnd, xStart:xEnd]

            features.append(np.max(chunk))

            features.append(np.mean(chunk))
            features.append(np.var(chunk))
            features.append(np.mean(edgeChunks))

            features.append(np.sum(edgeChunks > np.mean(edgeChunks)) / chunk.size)
            features.append(np.min(chunk))

            features.extend(np.percentile(chunk, [25, 50, 75]))

    features.extend([
        np.mean(original),
        np.min(original),
        np.var(original),
        np.max(original),

        np.mean(magnitude),
        np.max(magnitude),
        np.var(magnitude)
    ])
    if typeOfImage == 'face':
        rightHalf = np.fliplr(original[:, width//2:])

        leftHalf = original[:, :width//2]
        horizontalSymmetry = np.mean(np.abs(leftHalf - rightHalf))
        features.append(horizontalSymmetry)
        bottomHalf = np.flipud(original[height//2:, :])

        topHalf = original[:height//2, :]
        verticalSymmetry = np.mean(np.abs(topHalf - bottomHalf))
        features.append(verticalSymmetry)

    features = np.array(features)
    features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-10)

    return features.tolist()

def imageExtraction(img, typeOfImage='digit', gridUsage=True):
    if gridUsage:
        return imageProcessing(img, typeOfImage)

    pixels = np.array([float(pixel) for row in img for pixel in row])
    return (pixels / 255.0).tolist()
