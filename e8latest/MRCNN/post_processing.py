import pandas as pd
data = pd.read_excel("mrcnn_test.xlsx")
#target = data[['image_name','correct', 'gt_label', 'label', 'conf', 'iou']]
#result = target.groupby(['image_name','correct', 'gt_label', 'label'],as_index=False)[['conf', 'iou']].mean()
data['class_name'] = data['class_name'].fillna('None')
group_by = ['image_name','correct', 'gt_label', 'label', 'class_name']
result = data.groupby(group_by,as_index=False)[list(set(list(data.columns)) - set(group_by))].mean()
result.to_excel("mrcnn_result_test.xlsx")
