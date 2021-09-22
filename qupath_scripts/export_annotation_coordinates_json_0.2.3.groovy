//export annotation to geojson 
//qupath 0.2.3
//https://qupath.readthedocs.io/en/latest/docs/advanced/exporting_annotations.html

import qupath.lib.images.servers.ImageServer
import qupath.lib.io.GsonTools
//import groovy.json.JsonOutput
boolean prettyPrint = true



//def annotations = getAnnotationObjects()
//boolean prettyPrint = true
//def gson = GsonTools.getInstance(prettyPrint)
//println gson.toJson(annotations)

def outputDir = buildFilePath(PROJECT_BASE_DIR, 'jsons')
mkdirs(outputDir)

print "Exporting annotations to: " +  outputDir

def gson = GsonTools.getInstance(prettyPrint)
def json = gson.toJson(getCurrentServer())
def project = getProject()

c=0
for (entry in project.getImageList()) {
    def hierarchy = entry.readHierarchy()
    def annotations = hierarchy.getAnnotationObjects()
    def scn_name = entry.getImageName()

    
    // convert annotation data to json obj
    def data = gson.toJson(annotations)
    
    // write file
    def out_fp = buildFilePath(outputDir, scn_name + '.json')
    File file = new File(out_fp)
    new File(out_fp).write(data)
    
    print out_fp
    
    
    c++
}

print 'Done. Exported number of slides: ' + c