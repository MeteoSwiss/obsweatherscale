class Globals {
    // the library version
    static String version = 'latest'

    // the tag used when publishing documentation
    static String documentationTag = 'latest'
}

pipeline {
    agent {label 'podman'}

    parameters {
        booleanParam(name: 'RELEASE_BUILD', defaultValue: false, description: 'Creates and publishes a new release')
        booleanParam(name: 'PUBLISH_DOCUMENTATION', defaultValue: false, description: 'Publishes the generated documentation')
    }

    environment {
        SCANNER_HOME = tool name: 'Sonarqube-certs-PROD', type: 'hudson.plugins.sonar.SonarRunnerInstallation'

        PATH = "$workspace/.venv-mchbuild/bin:$HOME/tools/openshift-client-tools:$HOME/tools/trivy:$PATH"
        KUBECONFIG = "$workspace/.kube/config"
        HTTP_PROXY = 'http://proxy.meteoswiss.ch:8080'
        HTTPS_PROXY = 'http://proxy.meteoswiss.ch:8080'
        NO_PROXY = '.meteoswiss.ch,localhost'
    }

    options {
        gitLabConnection('CollabGitLab')

        // New jobs should wait until older jobs are finished
        disableConcurrentBuilds()
        // Discard old builds
        buildDiscarder(logRotator(artifactDaysToKeepStr: '7', artifactNumToKeepStr: '1', daysToKeepStr: '45', numToKeepStr: '10'))
        // Timeout the pipeline build after 1 hour
        timeout(time: 1, unit: 'HOURS')
    }

    stages {
        stage('Init') {
            steps {
                sh '''
                python -m venv .venv-mchbuild
                PIP_INDEX_URL=https://hub.meteoswiss.ch/nexus/repository/python-all/simple \
                    .venv-mchbuild/bin/pip install --upgrade mchbuild
                '''
                updateGitlabCommitStatus name: 'Build', state: 'running'
            }
        }

        stage('Test') {
            parallel {
                stage('python 3.10') {
                    steps {
                        sh 'mchbuild -s pythonImageName=\'"3.10"\' verify.unitWithoutCoverage'
                    }
                }
                stage('python 3.11') {
                    steps {
                        sh 'mchbuild -s pythonImageName=\'"3.11"\' verify.unitWithoutCoverage'
                    }
                }
                stage('python 3.12') {
                    steps {
                        sh 'mchbuild -s pythonImageName=\'"3.12"\' build test'
                        // lower the threshold once there are fewer issues, down to 10
                        recordIssues(qualityGates: [[threshold: 60, type: 'TOTAL', unstable: false]], tools: [myPy(pattern: 'mypy.log')])
                    }
                }
            }
            post {
                always {
                    junit keepLongStdio: true, testResults: 'junit*.xml'
                }
            }
        }

        stage('Scan') {
            steps {
                echo("---- DEPENDENCIES SECURITY SCAN ----")
                sh "mchbuild verify.securityScan"

                echo("---- SONARQUBE ANALYSIS ----")
                withSonarQubeEnv("Sonarqube-PROD") {
                    // fix source path in coverage.xml
                    // (required because coverage is calculated using podman which uses a differing file structure)
                    // https://stackoverflow.com/questions/57220171/sonarqube-client-fails-to-parse-pytest-coverage-results
                    sh "sed -i 's/\\/src\\/app-root/.\\//g' coverage.xml"
                    sh "${SCANNER_HOME}/bin/sonar-scanner"
                }

                echo("---- SONARQUBE QUALITY GATE ----")
                timeout(time: 1, unit: 'HOURS') {
                    // Parameter indicates whether to set pipeline to UNSTABLE if Quality Gate fails
                    // true = set pipeline to UNSTABLE, false = don't
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        stage('Release') {
            when { expression { params.RELEASE_BUILD } }
            steps {
                echo 'Build a wheel and publish'
                script {
                    withCredentials([string(credentialsId: 'python-mch-nexus-secret', variable: 'PYPIPASS')]) {
                        sh "PYPIUSER=python-mch mchbuild deploy.pypi"
                        Globals.version = sh(script: 'git describe --tags --abbrev=0', returnStdout: true).trim()
                        Globals.documentationTag = Globals.version
                        env.TAG_NAME = Globals.documentationTag
                    }

                    echo("---- PUBLISH DEPENDENCIES TO DEPENDENCY REGISTRY ----")
                    withCredentials([string(credentialsId: 'dependency-track-token-prod', variable: 'DTRACK_TOKEN')]) {
                       catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                            sh "mchbuild verify.publishSbom -s version=${Globals.version}"
                       }
                    }
                }
            }
        }

        stage('Publish Documentation') {
            when { expression { params.PUBLISH_DOCUMENTATION } }
            steps {
                withCredentials([string(credentialsId: 'documentation-main-prod-token',
                                        variable: 'DOC_TOKEN')]) {
                    sh """
                    mchbuild -s pythonImageName=3.12 -s deploymentEnvironment=prod \
                      -s docVersion=${Globals.documentationTag} deploy.docs
                    """
                }
            }
        }
    }

    post {
        aborted {
            updateGitlabCommitStatus name: 'Build', state: 'canceled'
        }
        failure {
            updateGitlabCommitStatus name: 'Build', state: 'failed'
            echo 'Sending email'
            emailext(subject: "${currentBuild.fullDisplayName}: ${currentBuild.currentResult}",
                attachLog: true,
                attachmentsPattern: 'generatedFile.txt',
                body: "Job '${env.JOB_NAME} #${env.BUILD_NUMBER}': ${env.BUILD_URL}",
                recipientProviders: [requestor(), developers()])
        }
        success {
            echo 'Build succeeded'
            updateGitlabCommitStatus name: 'Build', state: 'success'
        }
    }
}
