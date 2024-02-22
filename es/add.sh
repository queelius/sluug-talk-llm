curl -X POST "localhost:9200/articles/_doc/" -H 'Content-Type: application/json' -d'
{
  "title": "$1",
  "author": "$2",
  "publication_date": "$3"
}'
