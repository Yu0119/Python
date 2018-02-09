# -*- coding: utf-8 -*-
"""
 Brief: Base code for use sqlalchemy
"""
from __future__ import print_function
import sqlalchemy
import sqlalchemy.ext.declarative
import sqlalchemy.orm

# sqlite
# eng = sqlalchemy.create_engine('sqlite:///:memory:')
# SQL文ののCheck
eng = sqlalchemy.create_engine('sqlite:///:memory:', echo=True)
# Database作成
# eng = sqlalchemy.create_engine('sqlite:///test_sqlite')

# mysql
# eng = sqlalchemy.create_engine('mysql+pymysql:///test_mysql')

Decbase = sqlalchemy.ext.declarative.declarative_base()


# Item class
class Item(Decbase):
  """
    Brief:
      Item Database class
    Extends:
      Decbase
  """
  # Table名称
  __tablename__ = 'items'
  # カラムのID
  column_id = sqlalchemy.Column(
    sqlalchemy.Integer, primary_key=True, autoincrement=True
  )
  # ITEM名称(string)
  name = sqlalchemy.Column(sqlalchemy.String(30))
  price = sqlalchemy.Column(sqlalchemy.INTEGER())


if __name__ == '__main__':

  Decbase.metadata.create_all(eng)

  Session = sqlalchemy.orm.sessionmaker(bind=eng)
  session = Session()

  item1 = Item(name='ice-cream', price=150)
  session.add(item1)
  item2 = Item(name='candy', price=50)
  session.add(item2)

  # Add items ...

  # Commit set items
  session.commit()

  # Search all queries
  items = session.query(Item).all()

  for item in items:
    print(item.column_id, item.name, item.price)
  