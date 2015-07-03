import nose, warnings

ret = 0

# Run regression tests
with warnings.catch_warnings(record=True) as warns:
    result = nose.run(defaultTest='regression')

    # Handle failed test
    ret += int(not result)

    # Handle missing references
    for w in warns:
        warnings.showwarning(w.message, w.category, w.filename,
                             w.lineno, w.line)
    ret += len(warns)

# Run unit tests
result = nose.run(defaultTest='unit')
ret += int(not result)

exit(ret)
