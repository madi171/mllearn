from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
data = load_iris()
clf = DecisionTreeClassifier()
clf.fit(data.data, data.target)

# from sklearn.externals.six import StringIO
# dot_data = StringIO()
# f = export_graphviz(clf, out_file=dot_data)
# import pydot
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("output/iris.pdf")
print clf.score(data.data, data.target)