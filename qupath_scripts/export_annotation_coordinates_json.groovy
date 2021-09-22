import qupath.lib.images.servers.ImageServer
import qupath.lib.io.GsonTools
import groovy.json.JsonOutput
boolean prettyPrint = true

print "Exporting annotations to: " +  PROJECT_BASE_DIR

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
    def out_fp = buildFilePath(PROJECT_BASE_DIR, scn_name + '.json')
    File file = new File(out_fp)
    new File(out_fp).write(data)
    
    print out_fp
    
    
    c++
}

print 'Done. Exported number of slides: ' + c