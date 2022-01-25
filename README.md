# Heatmap

## Description 
Using video recordings from a paddel playingfield we are trying to map the locations players have walked. First we capture there movement and later on we convert the coordinates and plot them onto an image of the playingfield. This will give us insight in where people mostly walk and what areas are left unattended.

## Packages  
| Packages|
|--------|
| python 3.8 |
| OpenCV |
| datetime |
| numpy |
| collections - Defaultdict |
| pandas |
| imutils |

## Explanation of the app
1. The video gets imported and gets read frame by frame.
2. MobileNetSSD recognizes a human. 
3. BoundingBoxes get placed around the person. 
4. Non max surpression merges overlapping bbox into one.
5. The coordinates of the bbox get saved and converted using homography transformation. 
