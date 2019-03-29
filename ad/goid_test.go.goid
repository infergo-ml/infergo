package ad

import (
	"testing"
)

func TestGoID(t *testing.T) {
	id0 := goID()
	ch1 := make(chan int64)
	ch2 := make(chan int64)
	go func () {
		ch1<- goID()
	}()
	go func () {
		ch2<- goID()
	}()
	id1 := <-ch1
	id2 := <-ch2
	if id0 == id1 {
		t.Errorf("id0 and id1 must differ, but both are %v", id0)
	}
	if id1 == id2 {
		t.Errorf("id1 and id2 must differ, but got %v", id1)
	}
}
