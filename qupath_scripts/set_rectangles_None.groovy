// this is for setting all rectangular rois to None
// after this, run export mask script
// the roi mask will no longer cover the stain mask


def annotations = getAnnotationObjects()

for (annotation in annotations){

    roi = annotation.getROI()
    shape = roi.getShape()

    if (shape instanceof java.awt.geom.Rectangle2D$Double){
        annotation.setPathClass(getPathClass("Ignore"))
    }


}
