#Clientes --> Tabla1
#!PK (Llave Primaria):  customer_id

#Ventas --> Tabla2
#! PK (Llave Primaria):  order_id
#? ForeignKey: customer_id

# Relaciones:
#! 1) Cada venta...pertenece a un cliente: 1:1
#! 2) Cada cliente...puede tener muchas "ventas" 1:N

#todo: La llave primaria "customer_id" tiene que ir como "foranea" dentro de ventas por que
#todo: sin clientes.... no tengo ventas...

