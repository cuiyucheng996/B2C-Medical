@echo off
rem ============================================================
rem B2C business - Maven Wrapper (mvnw.cmd)
rem Downloads and caches Maven inside the project .m2 directory.
rem ============================================================
setlocal

if "%JAVA_HOME%"=="" (
    echo Error: JAVA_HOME not found. Please set JAVA_HOME to a JDK installation.
    exit /b 1
)
if not exist "%JAVA_HOME%\bin\java.exe" (
    echo Error: JAVA_HOME is set to an invalid directory: "%JAVA_HOME%"
    exit /b 1
)

rem Project base dir = directory containing this script (no trailing backslash)
set "MAVEN_PROJECTBASEDIR=%~dp0"
if "%MAVEN_PROJECTBASEDIR:~-1%"=="\" set "MAVEN_PROJECTBASEDIR=%MAVEN_PROJECTBASEDIR:~0,-1%"

rem Project-local cache: Maven dist + dependency repository live under .m2
if not defined MAVEN_USER_HOME set "MAVEN_USER_HOME=%MAVEN_PROJECTBASEDIR%\.m2"

set "MAVEN_JAVA_EXE=%JAVA_HOME%\bin\java.exe"
set "WRAPPER_JAR=%MAVEN_PROJECTBASEDIR%\.mvn\wrapper\maven-wrapper.jar"
if not exist "%WRAPPER_JAR%" (
    echo Error: maven-wrapper.jar not found at "%WRAPPER_JAR%"
    exit /b 1
)

set "WRAPPER_LAUNCHER=org.apache.maven.wrapper.MavenWrapperMain"

"%MAVEN_JAVA_EXE%" ^
  %MAVEN_OPTS% ^
  -Duser.home="%MAVEN_USER_HOME%" ^
  -Dmaven.repo.local="%MAVEN_USER_HOME%\repository" ^
  -classpath "%WRAPPER_JAR%" ^
  "-Dmaven.multiModuleProjectDirectory=%MAVEN_PROJECTBASEDIR%" ^
  %WRAPPER_LAUNCHER% -s "%MAVEN_PROJECTBASEDIR%\.mvn\settings.xml" %*

set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
