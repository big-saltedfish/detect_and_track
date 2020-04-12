#include<worker.h>


Worker wk("003.mp4", 20, 20, 8, 10);
int main(int argc, char **argv) {
    while (1) {
        cout << "ok" << endl;
        wk.update();
    }
}
