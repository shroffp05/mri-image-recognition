import pickle 

classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

filename = 'classes.pickle'
outfile = open(filename,'wb')

pickle.dump(classes,outfile)
outfile.close()