#!/bin/sh
# ============================================================
# B2C business - Maven Wrapper (mvnw)
# Downloads and caches Maven inside the project .m2 directory.
# ============================================================

# Resolve project base dir (absolute path of this script's directory)
MAVEN_PROJECTBASEDIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Project-local cache: Maven dist + dependency repository live under .m2
if [ -z "$MAVEN_USER_HOME" ]; then
  MAVEN_USER_HOME="$MAVEN_PROJECTBASEDIR/.m2"
  export MAVEN_USER_HOME
fi

if [ -z "$JAVACMD" ]; then
  if [ -n "$JAVA_HOME" ] && [ -x "$JAVA_HOME/bin/java" ]; then
    JAVACMD="$JAVA_HOME/bin/java"
  elif command -v java >/dev/null 2>&1; then
    JAVACMD=$(command -v java)
  fi
fi
if [ -z "$JAVACMD" ] || [ ! -x "$JAVACMD" ]; then
  echo "Error: JAVA_HOME is not set or invalid. Please set JAVA_HOME to a JDK installation." >&2
  exit 1
fi

WRAPPER_JAR="$MAVEN_PROJECTBASEDIR/.mvn/wrapper/maven-wrapper.jar"
if [ ! -f "$WRAPPER_JAR" ]; then
  echo "Error: maven-wrapper.jar not found at $WRAPPER_JAR" >&2
  exit 1
fi

exec "$JAVACMD" \
  $MAVEN_OPTS \
  -Duser.home="$MAVEN_USER_HOME" \
  -Dmaven.repo.local="$MAVEN_USER_HOME/repository" \
  -classpath "$WRAPPER_JAR" \
  "-Dmaven.multiModuleProjectDirectory=$MAVEN_PROJECTBASEDIR" \
  org.apache.maven.wrapper.MavenWrapperMain -s "$MAVEN_PROJECTBASEDIR/.mvn/settings.xml" "$@"
