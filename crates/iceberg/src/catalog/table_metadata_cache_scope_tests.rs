// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::{HashMap, HashSet};

use super::*;

#[test]
fn credential_context_selectors_each_distinguish() {
    let cases: [(&str, &str); 6] = [
        ("aws_access_key_id", "AKID-A"),
        ("profile_name", "prof-a"),
        (S3_ACCESS_KEY_ID, "s3-akid-a"),
        (S3_ASSUME_ROLE_ARN, "arn:aws:iam::1:role/r"),
        (S3_ASSUME_ROLE_EXTERNAL_ID, "ext-id-a"),
        (S3_ASSUME_ROLE_SESSION_NAME, "sess-a"),
    ];
    let mut contexts = HashSet::new();
    for (key, value) in cases {
        let props = HashMap::from([(key.to_string(), value.to_string())]);
        let context = CacheScope::credential_context_from_props(&props)
            .unwrap_or_else(|| panic!("selector {key} must derive a context"));
        assert!(context.contains(value), "{key} value must key the context");
        assert!(
            contexts.insert(context),
            "selector {key} must produce a distinct context"
        );
    }

    let props_a = HashMap::from([("aws_access_key_id".to_string(), "A".to_string())]);
    let props_b = HashMap::from([("aws_access_key_id".to_string(), "B".to_string())]);
    assert_ne!(
        CacheScope::credential_context_from_props(&props_a),
        CacheScope::credential_context_from_props(&props_b),
        "same selector with a different value must distinguish"
    );
}

#[test]
fn region_alone_never_derives_credential_context() {
    for key in ["region_name", "s3.region", "client.region"] {
        let props = HashMap::from([(key.to_string(), "us-east-1".to_string())]);
        assert!(
            CacheScope::credential_context_from_props(&props).is_none(),
            "{key} is not a credential selector"
        );
    }

    let props = HashMap::from([("region_name".to_string(), "us-east-1".to_string())]);
    let first = CacheScope::for_catalog("s3tables:arn:shared", None, &props);
    let second = CacheScope::for_catalog("s3tables:arn:shared", None, &props);
    assert_ne!(
        first, second,
        "region-only props must isolate every catalog instance"
    );
}

#[test]
fn secret_props_never_reach_context_or_debug() {
    let secrets = [
        ("aws_secret_access_key", "SECRET-ACCESS"),
        ("aws_session_token", "SECRET-TOKEN"),
        ("s3.secret-access-key", "SECRET-S3"),
        ("s3.session-token", "SECRET-ST"),
    ];
    let props: HashMap<String, String> = secrets
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    assert!(
        CacheScope::credential_context_from_props(&props).is_none(),
        "secrets are not credential selectors"
    );

    let mut props = props;
    props.insert("aws_access_key_id".to_string(), "AKID".to_string());
    let context = CacheScope::credential_context_from_props(&props)
        .expect("a real selector must derive a context");
    let scope = CacheScope::for_catalog("catalog:id", None, &props);
    let debug = format!("{scope:?}");
    for (_, secret) in secrets {
        assert!(
            !context.contains(secret),
            "secret value must never reach the credential context"
        );
        assert!(
            !debug.contains(secret),
            "secret value must never reach Debug output"
        );
    }
}

#[test]
fn scope_separation_identity_vs_credential_context() {
    let props = HashMap::from([("aws_access_key_id".to_string(), "A".to_string())]);
    let one = CacheScope::for_catalog("id:1", None, &props);
    let same = CacheScope::for_catalog("id:1", None, &props);
    assert_eq!(one, same, "identical identity and credentials share");

    let other_identity = CacheScope::for_catalog("id:2", None, &props);
    assert_ne!(one, other_identity, "identity alone must separate");

    let other_creds = CacheScope::for_catalog(
        "id:1",
        None,
        &HashMap::from([("aws_access_key_id".to_string(), "B".to_string())]),
    );
    assert_ne!(one, other_creds, "credential difference must separate");

    let explicit = CacheScope::for_catalog("id:1", Some("ctx-explicit".to_string()), &props);
    assert_ne!(one, explicit, "an explicit context must separate");
}
