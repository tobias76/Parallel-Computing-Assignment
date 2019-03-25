#include <iostream>

#pragma once
class Utilities
{
public:
	Utilities();
	~Utilities();

	ifstream &File;
	void padVector();
private:
	// Current line in the file.
	string currentLine;

};

