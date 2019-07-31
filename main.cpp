#include "Final_Project.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QCoreApplication::addLibraryPath(".");
	QApplication a(argc, argv);
	Final_Project w;
	w.show();
	return a.exec();
}
