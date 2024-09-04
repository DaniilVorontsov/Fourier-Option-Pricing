
void VG(bool errorFlag, bool timeFlag);
void CGMY(bool errorFlag, bool timeFlag);


int main()
{
	bool errorFlag = true; //if true, write "CGMY_Errors.csv" and "VG_Errors.csv" files
	bool timeFlag = false; //if true, write "CGMY_Time.csv" and "VG_Time.csv" files

	//VG(errorFlag, timeFlag);
	CGMY(errorFlag, timeFlag);

	return 0;
}