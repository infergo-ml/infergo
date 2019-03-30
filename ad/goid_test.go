package ad

import (
	"testing"
)

func TestGoid(t *testing.T) {
	id0 := goid()
	ch1 := make(chan int64)
	ch2 := make(chan int64)
	go func() {
		ch1 <- goid()
	}()
	go func() {
		ch2 <- goid()
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
