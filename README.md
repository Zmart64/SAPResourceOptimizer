# Distributed_Systems_Project

### 1. Exercise - Alibaba Clusterdata (until 13th May)
#### The data recides in the home directory of our server and should be used from there. Create a symlink like this:
```console
ln -s /home/project/data/clusterdata/cluster-trace-microservices-v2022/data /home/{your_user}/Distributed_Systems_Project/
```

- on alibaba clusterdata
	- download 1 hour first for overview
	- https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022
	- smaller data set (one or two days instead of 2 weeks)
- are micro services anonymous?
- classify micro services via callgraph
- come up with own labels
	- clustering, kNN
	- when is micro service called?
		- by other micro service
	- identify same micro service
- based on classification (anwendungen, micro services), predict ressources 
		- cpu, ram, latency
- github