import RaviartThomas, shapes

shape = 2
degree = 0

U = RaviartThomas.RTSpace(shape,degree)

pts = shapes.make_points(shape,shape,0,degree+3)

print pts
print U.tabulate( pts )