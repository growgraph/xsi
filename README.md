# xsi
rXiv Semantic Impact (XSI)


to restore db from an [XSI snapshot](https://zenodo.org/records/14866949)

```shell
arangorestore --server.endpoint tcp://ip:port --server.username root --server.password 123 --input-directory path  ./matkg --server.database "matkg" --create-database true
```
