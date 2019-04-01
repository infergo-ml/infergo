package ad

import (
	"testing"
)

func TestGoid(t *testing.T) {
	if !MTSafeOn() {
		return
	}
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

func TestAtleast(t *testing.T) {
	for _, c := range []struct{
		version string
		major, minor, patch int
		answer bool
	} {
		{"go1.2", 1, 1, 0, true},
		{"go1.2", 1, 10, 0, false},
		{"go1.2beta3", 1, 2, 0, true},
		{"go1.2beta3", 1, 2, 3, false},
		{"weekly.2006-01-02", 1, 10, 0, true},
	} {
		answer := atleast(c.version, c.major, c.minor, c.patch)
		if answer != c.answer {
			t.Errorf("wrong answer for version=%v, major=%v, minor=%v, patch=%v: " +
				"got %v, want %v",
				c.version, c.major, c.minor, c.patch,
				answer, c.answer)
		}
	}
}
