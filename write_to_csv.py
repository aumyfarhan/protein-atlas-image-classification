output_file='submission2.csv'

# appending the test data sample id and predicted data sample class label in a list
table=[]
id_name=str('id')
pred_name=str('predicted')

table.append([id_name,pred_name])
 
for i,j in enumerate(test_image_id_list):
    y=sub_label[i]
    ystr=[str(xx) for xx in y]
    ycstr=' '.join(ystr) 
    table.append([j,ycstr])


# creating a csv file of data sample id and their corresponding  class label 
with open(output_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in table:
        writer.writerow(val)