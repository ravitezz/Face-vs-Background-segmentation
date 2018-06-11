# Face-vs-Background-segmentation
Face vs background segmentation

This project is about segmenting face from background using Generative model. We make use of training images to obtain the likelihood and prior distributions, later we use these values to infer the facial regions using  test images. We make use of RGB color space for this purpose.

Algorithmic Approach:
1. We make use of Generative model for detect facial region.
2. We take training data set and generate rectangular region around face. We assign a value of 1 for those pixels.
3. We calculate prior based on the number of pixels in facial region and the total number of pixels (1920x1080)
4. Later we compute likelihood values for these facial images by taking r,g,b histogram and incrementing the magnitude of the histogram.
5. Later we marginalize the histogram to probability distribution.
6. Later inference is carried out using posterior , the product of likelihood and prior, such that if the product of likelihood of being a facial pixel and its prior is more than non-facial pixel, it will be inferred as facial pixel. 
7. Parameters like True positive, True negative, false positive, false negatives are calculated to determine accuracy precision, recall


