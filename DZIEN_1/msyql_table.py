import mysql.connector

db = mysql.connector.connect(user='root',password='abc123',host='127.0.0.1',port=3306, database="dbtestowa")

cursorObject = db.cursor()
# table_student = "CREATE TABLE student(name varchar(255),nr_alb int);"
# cursorObject.execute(table_student)
#
# sql_st_data = "INSERT INTO student(name,nr_alb) values(%s,%s)"
# val = ("Jan",2345)
# cursorObject.execute(sql_st_data,val)
# db.commit()
# print(cursorObject.rowcount,"element osadzony")

insert_list = "INSERT INTO student(name,nr_alb) values(%s,%s)"
val = [
    ("Maria",5343),
    ("Marian",6565),
    ("Olaf",656),
    ("Robert",643),
    ("Henio",5657465),
    ("Gienio",756),
    ("Tadek",1232),
]
cursorObject.executemany(insert_list,val)
db.commit()

print(cursorObject.rowcount,"element/ów dodano")
db.close()
