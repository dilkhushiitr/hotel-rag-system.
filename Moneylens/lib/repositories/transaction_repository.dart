import 'package:moneylens/core/constants/app_constants.dart';
import 'package:moneylens/database/database_helper.dart';
import 'package:moneylens/models/transaction_model.dart';

class TransactionRepository {
  TransactionRepository._();
  static final TransactionRepository instance = TransactionRepository._();

  final _db = DatabaseHelper.instance;
  final String _table = AppConstants.tableTransactions;

  Future<bool> saveTransaction(TransactionModel tx) async {
    final rows = await _db.insert(_table, tx.toMap());
    return rows > 0;
  }

  Future<int> saveAll(List<TransactionModel> transactions) async {
    int saved = 0;
    for (final tx in transactions) {
      final inserted = await saveTransaction(tx);
      if (inserted) saved++;
    }
    return saved;
  }

  Future<List<TransactionModel>> getAll() async {
    final rows = await _db.query(_table, orderBy: 'date DESC');
    return rows.map(TransactionModel.fromMap).toList();
  }

  Future<List<TransactionModel>> getToday() async {
    final now = DateTime.now();
    final start = DateTime(now.year, now.month, now.day);
    final end = start.add(const Duration(days: 1));
    return _getByDateRange(start, end);
  }

  Future<List<TransactionModel>> getThisWeek() async {
    final now = DateTime.now();
    final start = now.subtract(Duration(days: now.weekday - 1));
    final weekStart = DateTime(start.year, start.month, start.day);
    final weekEnd = weekStart.add(const Duration(days: 7));
    return _getByDateRange(weekStart, weekEnd);
  }

  Future<List<TransactionModel>> getThisMonth() async {
    final now = DateTime.now();
    final start = DateTime(now.year, now.month, 1);
    final end = DateTime(now.year, now.month + 1, 1);
    return _getByDateRange(start, end);
  }

  Future<List<TransactionModel>> getByCategory(String category) async {
    final rows = await _db.query(
      _table,
      where: 'category = ?',
      whereArgs: [category],
      orderBy: 'date DESC',
    );
    return rows.map(TransactionModel.fromMap).toList();
  }

  Future<List<TransactionModel>> search(String query) async {
    final q = '%${query.toLowerCase()}%';
    final rows = await _db.rawQuery(
      '''
      SELECT * FROM $_table
      WHERE LOWER(merchant) LIKE ?
         OR LOWER(category) LIKE ?
         OR CAST(amount AS TEXT) LIKE ?
      ORDER BY date DESC
      ''',
      [q, q, q],
    );
    return rows.map(TransactionModel.fromMap).toList();
  }

  Future<List<TransactionModel>> getRecent({int limit = 10}) async {
    final rows = await _db.query(
      _table,
      orderBy: 'date DESC',
      limit: limit,
    );
    return rows.map(TransactionModel.fromMap).toList();
  }

  Future<bool> hashExists(String hash) async {
    final rows = await _db.query(
      _table,
      where: 'smsHash = ?',
      whereArgs: [hash],
      limit: 1,
    );
    return rows.isNotEmpty;
  }

  Future<double> sumDebits({
    required DateTime from,
    required DateTime to,
  }) async {
    final rows = await _db.rawQuery(
      '''
      SELECT COALESCE(SUM(amount), 0) AS total
      FROM $_table
      WHERE transactionType = 'debit'
        AND date >= ?
        AND date < ?
      ''',
      [from.millisecondsSinceEpoch, to.millisecondsSinceEpoch],
    );
    return (rows.first['total'] as num).toDouble();
  }

  Future<List<CategorySummary>> getCategoryTotalsThisMonth() async {
    final now = DateTime.now();
    final start = DateTime(now.year, now.month, 1);
    final end = DateTime(now.year, now.month + 1, 1);

    final rows = await _db.rawQuery(
      '''
      SELECT
        category,
        SUM(amount) AS totalAmount,
        COUNT(*) AS transactionCount
      FROM $_table
      WHERE transactionType = 'debit'
        AND date >= ?
        AND date < ?
      GROUP BY category
      ORDER BY totalAmount DESC
      ''',
      [start.millisecondsSinceEpoch, end.millisecondsSinceEpoch],
    );
    return rows.map(CategorySummary.fromMap).toList();
  }

  Future<String> getTopMerchantThisMonth() async {
    final now = DateTime.now();
    final start = DateTime(now.year, now.month, 1);
    final end = DateTime(now.year, now.month + 1, 1);

    final rows = await _db.rawQuery(
      '''
      SELECT merchant, SUM(amount) AS total
      FROM $_table
      WHERE transactionType = 'debit'
        AND date >= ?
        AND date < ?
      GROUP BY merchant
      ORDER BY total DESC
      LIMIT 1
      ''',
      [start.millisecondsSinceEpoch, end.millisecondsSinceEpoch],
    );
    if (rows.isEmpty) return '—';
    return rows.first['merchant'] as String;
  }

  Future<void> deleteAll() async {
    await _db.delete(_table);
  }

  Future<void> deleteById(String id) async {
    await _db.delete(_table, where: 'id = ?', whereArgs: [id]);
  }

  Future<List<TransactionModel>> _getByDateRange(
    DateTime from,
    DateTime to,
  ) async {
    final rows = await _db.query(
      _table,
      where: 'date >= ? AND date < ?',
      whereArgs: [from.millisecondsSinceEpoch, to.millisecondsSinceEpoch],
      orderBy: 'date DESC',
    );
    return rows.map(TransactionModel.fromMap).toList();
  }
}
