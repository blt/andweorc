#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("Calling the fopen() function...\n");

    FILE *fd = fopen("test.txt","r");
    if (!fd) {
        printf("fopen() returned NULL\n");
        exit(1);
    }

    printf("fopen() succeeded\n");

    exit(0);
}
