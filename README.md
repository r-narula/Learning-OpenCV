# OPEN-CV Notes

For destroying images with the a key ...
```
cv2.imshow('img',imgElon)
if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
```

```
Haar Cascade Notes-> 
It is used for object detection in the image.
Read Voils Jones Algorithm for Face Detection ..

#### Importing the Haar Model #####
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_alt2.xml')
```
Haar Cascade is a very useful classifier that is being used for detecting the Faces..
Ueful Link for Voila Jones Algorithm
```earning Tutorialwww.dezyre.com › data-science-in-python-tutorial › pr...
Principal Component Analysis Tutorial. As you get ready to work on a PCA based project, we thought it will be helpful to give you ready-to-use code snippets. if ...

A Step by Step Explanation of Principal Component Analysisbuiltin.com › data-science › step-step-explanation-princ...
Sep 4, 2019 — The purpose of this post is to provide a complete and simplified explanation of Principal Component Analysis, and especially to answer how it ...

When and where do we use PCA? - Quorawww.quora.com › When-and-where-do-we-use-PCA
Nov 30, 2014 — PCA is an unsupervised linear dimensionality reduction algorithm to find a more
https://www.youtube.com/watch?v=x41KFOFGnUE&ab_channel=GlobalSoftwareSupport
```
It finds the Haar features and then we need just to mark the face.


```
PCA Is a Dimensionality Reduction technique ..
Reduce the number of columns
Why do we need Dimensionality Reduction --> 

eg ->                 __
Size                    |
Number of Rooms         |-------->>>> Lumped Together to form Size 
Number of Bathrooms   __|
While reducing the columns too we are capturing information adequeately..


```             
