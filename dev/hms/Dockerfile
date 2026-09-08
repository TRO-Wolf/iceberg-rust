# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM curlimages/curl:8.16.0@sha256:463eaf6072688fe96ac64fa623fe73e1dbe25d8ad6c34404a669ad3ce1f104b6 AS artifacts

ENV HADOOP_AWS_VERSION=3.1.0
ENV HADOOP_AWS_SHA256=a18508b9348af095ea41301e439354dbd449e304ac44c6885b2b4fe78de88126
ENV AWS_SDK_BUNDLE_VERSION=1.11.271
ENV AWS_SDK_BUNDLE_SHA256=faf78ac4880f56cf52791d84ec1068ce7c66acc4295d580a726104b734c01fcd

WORKDIR /artifacts

USER root

RUN curl --fail --show-error --location --proto '=https' --proto-redir '=https' \
        "https://repo.maven.apache.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar" \
        --output "hadoop-aws-${HADOOP_AWS_VERSION}.jar.download" && \
    echo "${HADOOP_AWS_SHA256}  hadoop-aws-${HADOOP_AWS_VERSION}.jar.download" | sha256sum --check && \
    mv "hadoop-aws-${HADOOP_AWS_VERSION}.jar.download" "hadoop-aws-${HADOOP_AWS_VERSION}.jar" && \
    curl --fail --show-error --location --proto '=https' --proto-redir '=https' \
        "https://repo.maven.apache.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_BUNDLE_VERSION}/aws-java-sdk-bundle-${AWS_SDK_BUNDLE_VERSION}.jar" \
        --output "aws-java-sdk-bundle-${AWS_SDK_BUNDLE_VERSION}.jar.download" && \
    echo "${AWS_SDK_BUNDLE_SHA256}  aws-java-sdk-bundle-${AWS_SDK_BUNDLE_VERSION}.jar.download" | sha256sum --check && \
    mv "aws-java-sdk-bundle-${AWS_SDK_BUNDLE_VERSION}.jar.download" "aws-java-sdk-bundle-${AWS_SDK_BUNDLE_VERSION}.jar"

FROM apache/hive:3.1.3

ENV AWSSDK_VERSION=2.20.18
ENV HADOOP_VERSION=3.1.0

USER root

COPY --from=artifacts --chown=hive:hive --chmod=0644 /artifacts/hadoop-aws-3.1.0.jar /opt/hive/lib/hadoop-aws-3.1.0.jar
COPY --from=artifacts --chown=hive:hive --chmod=0644 /artifacts/aws-java-sdk-bundle-1.11.271.jar /opt/hive/lib/aws-java-sdk-bundle-1.11.271.jar

COPY core-site.xml /opt/hadoop/etc/hadoop/core-site.xml

USER hive
