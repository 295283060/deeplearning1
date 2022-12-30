import csv

def get_csv(data):
    f = open('data.csv', 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(f)
    csv_write.writerow(['image_id','is_male'])
    for item in data:
        csv_write.writerow([item[0], item[1]])
    f.close()