.PHONY: build test vet clean

build:
	go build -o ./bin/egent-jobs .

test:
	go test ./...

vet:
	go vet ./...

clean:
	rm -rf ./bin
