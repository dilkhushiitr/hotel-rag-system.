import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'package:moneylens/core/constants/app_constants.dart';

class DatabaseHelper {
  DatabaseHelper._();
  static final DatabaseHelper instance = DatabaseHelper._();

  Database? _db;

  Future<Database> get database async {
    _db ??= await _initDatabase();
    return _db!;
  }

  // ─── Init ────────────────────────────────────────────────────────────────

  Future<Database> _initDatabase() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, AppConstants.dbName);

    return openDatabase(
      path,
      version: AppConstants.dbVersion,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }

  // ─── Schema Creation ─────────────────────────────────────────────────────

  Future<void> _onCreate(Database db, int version) async {
    await _createTransactionsTable(db);
    await _createMerchantsTable(db);
    await _createSettingsTable(db);
    await _seedDefaultMerchants(db);
  }

  Future<void> _createTransactionsTable(Database db) async {
    await db.execute('''
      CREATE TABLE ${AppConstants.tableTransactions} (
        id          TEXT    PRIMARY KEY,
        amount      REAL    NOT NULL,
        merchant    TEXT    NOT NULL,
        category    TEXT    NOT NULL,
        bank        TEXT    NOT NULL DEFAULT '',
        transactionType TEXT NOT NULL DEFAULT 'debit',
        date        INTEGER NOT NULL,
        smsBody     TEXT    NOT NULL DEFAULT '',
        smsHash     TEXT    NOT NULL UNIQUE,
        createdAt   INTEGER NOT NULL
      )
    ''');

    // Index for fast date-range queries (dashboard)
    await db.execute('''
      CREATE INDEX idx_transactions_date
      ON ${AppConstants.tableTransactions} (date DESC)
    ''');

    // Index for category filtering
    await db.execute('''
      CREATE INDEX idx_transactions_category
      ON ${AppConstants.tableTransactions} (category)
    ''');

    // Index for merchant filtering
    await db.execute('''
      CREATE INDEX idx_transactions_merchant
      ON ${AppConstants.tableTransactions} (merchant)
    ''');
  }

  Future<void> _createMerchantsTable(Database db) async {
    await db.execute('''
      CREATE TABLE ${AppConstants.tableMerchants} (
        merchant    TEXT PRIMARY KEY,
        category    TEXT NOT NULL
      )
    ''');
  }

  Future<void> _createSettingsTable(Database db) async {
    await db.execute('''
      CREATE TABLE ${AppConstants.tableSettings} (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
      )
    ''');
  }

  // ─── Seed Data ───────────────────────────────────────────────────────────

  Future<void> _seedDefaultMerchants(Database db) async {
    const merchants = {
      // Food
      'swiggy': 'Food', 'zomato': 'Food', 'dominos': 'Food',
      'mcdonalds': 'Food', 'kfc': 'Food', 'subway': 'Food',
      'dunkin': 'Food', 'starbucks': 'Food',
      // Grocery
      'blinkit': 'Grocery', 'zepto': 'Grocery', 'bigbasket': 'Grocery',
      'dmart': 'Grocery', 'jiomart': 'Grocery',
      // Travel
      'uber': 'Travel', 'ola': 'Travel', 'rapido': 'Travel',
      'irctc': 'Travel', 'makemytrip': 'Travel', 'redbus': 'Travel',
      // Shopping
      'amazon': 'Shopping', 'flipkart': 'Shopping', 'myntra': 'Shopping',
      'ajio': 'Shopping', 'meesho': 'Shopping', 'nykaa': 'Shopping',
      // Bills
      'airtel': 'Bills', 'jio': 'Bills', 'vodafone': 'Bills',
      'bescom': 'Bills', 'tata power': 'Bills',
      // Entertainment
      'netflix': 'Entertainment', 'spotify': 'Entertainment',
      'bookmyshow': 'Entertainment', 'hotstar': 'Entertainment',
      'disney': 'Entertainment',
      // Medical
      'apollo': 'Medical', 'pharmaeasy': 'Medical',
      'netmeds': 'Medical', '1mg': 'Medical',
      // Fuel
      'hpcl': 'Fuel', 'bpcl': 'Fuel', 'iocl': 'Fuel', 'shell': 'Fuel',
    };

    final batch = db.batch();
    for (final entry in merchants.entries) {
      batch.insert(AppConstants.tableMerchants, {
        'merchant': entry.key,
        'category': entry.value,
      });
    }
    await batch.commit(noResult: true);
  }

  // ─── Migrations ──────────────────────────────────────────────────────────

  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    // Future migrations go here
    // e.g. if (oldVersion < 2) { await db.execute('ALTER TABLE ...'); }
  }

  // ─── Generic CRUD ────────────────────────────────────────────────────────

  Future<int> insert(String table, Map<String, dynamic> data) async {
    final db = await database;
    return db.insert(table, data, conflictAlgorithm: ConflictAlgorithm.ignore);
  }

  Future<List<Map<String, dynamic>>> query(
    String table, {
    String? where,
    List<dynamic>? whereArgs,
    String? orderBy,
    int? limit,
    int? offset,
  }) async {
    final db = await database;
    return db.query(
      table,
      where: where,
      whereArgs: whereArgs,
      orderBy: orderBy,
      limit: limit,
      offset: offset,
    );
  }

  Future<int> update(
    String table,
    Map<String, dynamic> data, {
    required String where,
    required List<dynamic> whereArgs,
  }) async {
    final db = await database;
    return db.update(table, data, where: where, whereArgs: whereArgs);
  }

  Future<int> delete(
    String table, {
    String? where,
    List<dynamic>? whereArgs,
  }) async {
    final db = await database;
    return db.delete(table, where: where, whereArgs: whereArgs);
  }

  Future<List<Map<String, dynamic>>> rawQuery(
    String sql, [
    List<dynamic>? args,
  ]) async {
    final db = await database;
    return db.rawQuery(sql, args);
  }

  Future<int> rawDelete(String sql, [List<dynamic>? args]) async {
    final db = await database;
    return db.rawDelete(sql, args);
  }

  // ─── Utility ─────────────────────────────────────────────────────────────

  /// Wipe everything — used in Settings > Delete Data
  Future<void> deleteAllData() async {
    final db = await database;
    await db.delete(AppConstants.tableTransactions);
    await db.delete(AppConstants.tableSettings);
    // Keep merchant mappings — they are app config, not user data
  }

  Future<void> close() async {
    final db = _db;
    if (db != null) {
      await db.close();
      _db = null;
    }
  }
}
